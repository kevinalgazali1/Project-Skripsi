<?php

namespace App\Http\Controllers;

use Carbon\Carbon;
use App\Models\User;
use App\Models\Loker;
use App\Models\DaftarLoker;
use Illuminate\Support\Str;
use App\Exports\LokerExport;
use Illuminate\Http\Request;
use App\Exports\PesertaLokerExport;
use Illuminate\Support\Facades\Auth;
use Maatwebsite\Excel\Facades\Excel;
use App\Http\Requests\LokerPostRequest;

class LokerController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index(Request $request)
    {
        $user = User::with('mitraProfile')->find(Auth::id());

        $query = Loker::with(['daftarLoker.user'])
            ->withCount('daftarLoker')
            ->where('user_id', $user->id)
            ->orderBy('created_at', 'desc');

        // Filter pencarian
        if ($request->filled('nama_perusahaan')) {
            $query->where('nama_perusahaan', 'like', '%' . $request->nama_perusahaan . '%');
        }

        if ($request->filled('lokasi')) {
            $query->where('lokasi', 'like', '%' . $request->lokasi . '%');
        }

        if ($request->filled('tipe')) {
            $query->where('tipe', $request->tipe);
        }

        if ($request->filled('status')) {
            $query->where('status', $request->status);
        }

        $lokerMitraSendiri = $query->paginate(5)->withQueryString()->through(function ($s) {
            $pesertas = collect($s->daftarLoker);

            $s->jumlah_asal_sekolah = $pesertas->pluck('asal_sekolah')->unique()->count();
            $s->jumlah_jurusan = $pesertas->pluck('jurusan')->unique()->count();
            $s->jumlah_kuliah = $pesertas->where('status_saat_ini', 'Kuliah')->count();
            $s->jumlah_bekerja_dan_usaha = $pesertas->whereIn('status_saat_ini', ['Bekerja', 'Wirausaha'])->count();
            $s->jumlah_tidak_bekerja = $pesertas->where('status_saat_ini', 'Tidak Bekerja')->count();

            return $s;
        });

        // Ambil loker mitra lain (tanpa pagination)
        $lokerMitraLain = Loker::where('user_id', '!=', $user->id)
            ->latest()
            ->limit(6)
            ->get();

        return view('mitra.loker', compact(
            'lokerMitraSendiri',
            'lokerMitraLain',
            'user'
        ));
    }

    /**
     * Show the form for creating a new resource.
     */
    public function create()
    {
        //
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(LokerPostRequest $request)
    {
        try {
            $user_id = $request->user()->id;
            $mitra_id = $request->user()->id;

            // Format gaji jadi satu string
            $gaji = 'Rp ' . number_format($request->gaji_min, 0, ',', '.') .
                ' - Rp ' . number_format($request->gaji_max, 0, ',', '.');

            // Upload gambar
            $file = $request->file('gambar');
            $file_name = $file->hashName();
            $file->move(public_path('storage/assets/loker'), $file_name);

            // Simpan data ke database
            Loker::create([
                'nama_perusahaan' => $request->nama_perusahaan,
                'deskripsi' => $request->deskripsi,
                'posisi' => $request->posisi,
                'lokasi' => $request->lokasi,
                'tipe' => $request->tipe,
                'pendidikan' => $request->pendidikan,
                'gaji' => $gaji,
                'gambar' => 'assets/loker/' . $file_name,
                'user_id' => $user_id,
                'mitra_id' => $mitra_id,
                'skills' => $request->skills,
            ]);

            return redirect()->back()->with('success', 'Lowongan Kerja Berhasil Dibuat');
        } catch (\Exception $e) {
            return redirect()->back()->with('danger', 'Lowongan Kerja Gagal Dibuat');
        }
    }


    /**
     * Display the specified resource.
     */
    public function show($loker_id)
    {
        $loker = Loker::with('daftarLoker.user')->findOrFail($loker_id);
        $pelamar = DaftarLoker::where('loker_id', $loker_id)->get();
    }

    /**
     * Show the form for editing the specified resource.
     */
    public function edit(string $id)
    {
        //
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, string $id)
    {
        $loker = Loker::findOrFail($id);
        $loker->nama_perusahaan = $request->nama_perusahaan;
        $loker->deskripsi = $request->deskripsi;
        $loker->posisi = $request->posisi;
        $loker->lokasi = $request->lokasi;
        $loker->tipe = $request->tipe;
        $loker->pendidikan = $request->pendidikan;
        $loker->status = $request->status;
        $loker->skills = $request->skills;

        // Format gaji jadi satu string
        $loker->gaji = 'Rp ' . number_format($request->gaji_min, 0, ',', '.') .
            ' - Rp ' . number_format($request->gaji_max, 0, ',', '.');

        if ($request->hasFile('gambar')) {
            // 🔥 Hapus file lama jika ada
            $oldPath = public_path('storage/' . $loker->gambar);
            if (file_exists($oldPath)) {
                unlink($oldPath);
            }

            // 📥 Upload file baru
            $file = $request->file('gambar');
            $file_name = $file->hashName();
            $file->move(public_path('storage/assets/loker'), $file_name);

            // 💾 Simpan path baru
            $loker->gambar = 'assets/loker/' . $file_name;
        }

        $loker->save();

        return redirect()->back()->with('success', 'Lowongan Kerja berhasil diperbarui.');
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(string $id)
    {
        $loker = Loker::findOrFail($id);
        if ($loker->user_id != Auth::id()) {
            abort(403);
        }
        $loker->delete();
        return redirect()->back()->with('success', 'Lowongan Kerja berhasil dihapus.');
    }

    public function getPeserta($id)
    {
        $peserta = DaftarLoker::where('loker_id', $id)->get();
        return response()->json($peserta);
    }

    public function exportPesertaExcel($id)
    {
        $loker = Loker::findOrFail($id);
        $filename = 'daftar_peserta_' . Str::slug($loker->posisi) . '.xlsx';

        return Excel::download(new PesertaLokerExport($id), $filename);
    }

    public function exportLoker(Request $request)
    {
        return Excel::download(new LokerExport($request), 'data_loker.xlsx');
    }
}
